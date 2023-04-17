import logging
import ssl
import sys

import click
import pika

from .basic_consumer import BasicConsumer, ReconnectingConsumer
from .handler import (
    CompositeHandler,
    DatamartDownloadHandler,
    DatamartMessageHandler,
    PrintHandler,
    SimpleDownloader,
)

LOGGER = logging.getLogger(__name__)

SSL_CONTEXT = ssl.SSLContext()
DATAMART_CONNECTION_PARAMS = pika.ConnectionParameters(
    host="dd.weather.gc.ca",
    port="5671",
    credentials=pika.PlainCredentials("anonymous", "anonymous"),
    virtual_host="/",
    ssl_options=pika.SSLOptions(context=SSL_CONTEXT),
)
DATAMART_EXCHANGE = "xpublic"
# Permissions seem to be allowed only when queues start with q_anonymous.
# Also, they like it when we add our company name to the queue.
# See https://eccc-msc.github.io/open-data/msc-datamart/amqp_fr/.
DATAMART_QUEUE_PREFIX = "q_anonymous.CRIM."

# This prefixes all the routing keys to connect to the messages.
# This was reverse engineered from Sarracenia. https://github.com/MetPX/Sarracenia.
DATAMART_ROUTING_KEY_PREFIX = "v02.post."


class DatamartConsumer(BasicConsumer):
    """AMQP message consumer specialized to the MSC Datamart.
    Example usage:::

        consumer = DatamartConsumer('model_gem_global.15km.grib2.lat_lon.#', 'gdps')
        consumer.run()
    """

    def __init__(self, topic, queue, handler):
        routing_key = DATAMART_ROUTING_KEY_PREFIX + topic
        queue = DATAMART_QUEUE_PREFIX + queue

        LOGGER.debug("Using routing key %s" % routing_key)
        LOGGER.debug("Using queue %s" % queue)

        super().__init__(
            DATAMART_CONNECTION_PARAMS,
            DATAMART_EXCHANGE,
            routing_key,
            queue,
            handler=handler,
            declare_exchange=False,
        )


@click.command()
@click.option(
    "--topic",
    required=True,
    default="model_gem_global.15km.grib2.lat_lon.#",
    help=(
        "The topic string of the messages you want to listen to. Corresponds the the directory of the "
        "files on the datamart. Ex: model_gem_global.15km.grib2.lat_lon.#"
    ),
)
@click.option("--queue", required=True, help="Name of the AMQP queue to use.")
@click.option(
    "--download-dir", required=True, default=".", help="Where to download the files."
)
@click.option("--verbose", is_flag=True)
def cli(topic, queue, download_dir, verbose):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    handler = CompositeHandler([PrintHandler()])

    def consumer_factory():
        return DatamartConsumer(topic, queue, handler=handler)

    consumer = ReconnectingConsumer(consumer_factory)
    consumer.run()


if __name__ == "__main__":
    cli()
